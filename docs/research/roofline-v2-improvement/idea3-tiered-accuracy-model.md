# Tiered Accuracy Model for MFU-Based LLM Inference Simulation

## Executive Summary

BLIS's roofline v2 model has higher prediction error than expected. We propose the **Tiered Accuracy Model (TAM)** — a three-tier framework that progressively improves accuracy with increasing calibration effort:

| Tier | What you need | What you get | Recalibration |
|------|--------------|--------------|---------------|
| **0** | Nothing — physics and execution semantics only | < 20% E2E MAPE | Never |
| **1** | One GuideLLM trace (≥100 requests) | < 15% E2E MAPE | When vLLM version changes |
| **2** | Automated GPU micro-benchmarks | < 12% E2E MAPE | When hardware changes |

Tier 0 requires zero fitted parameters — it corrects six structural mismatches between what the model assumes and how GPUs and vLLM actually behave. Tier 1 extracts version-specific overhead from a single client-side trace. Tier 2 replaces uniform bandwidth assumptions with measured per-access-pattern values.

**Constraint**: No server-side traces or vLLM instrumentation. All corrections derive from client-side metrics, analytical modeling, or offline benchmarks.

The six error patterns addressed by Tier 0 are detailed in [Section 1](#1-six-hypotheses-observable-error-patterns). Recommended execution order: H1 and H2 first (largest expected impact), then H3 and H5, then H4 and H6.

---

## Abstract

BLIS's roofline v2 model produces higher prediction errors than expected. We identify six **observable error patterns** in the simulator's output — systematic biases that can be detected purely by comparing predicted vs. measured latency across different workload regimes. Each pattern points to a structural mismatch between the model's assumptions and the physics of GPU execution or vLLM serving. We propose the **Tiered Accuracy Model (TAM)**, a three-tier correction framework organized by calibration cost: Tier 0 (zero-config), Tier 1 (one-trace calibration), Tier 2 (hardware characterization).

**Constraint**: No server-side traces, vLLM internal instrumentation, or framework-specific logging. All improvements derive from client-side metrics (GuideLLM), analytical modeling, and offline kernel benchmarks.

---

## 1. Six Hypotheses: Observable Error Patterns

Each hypothesis is a testable claim about what the simulator gets wrong, stated in terms of observable metrics. The proposed mechanism explains *why* the error exists. Implementation details and code references appear in the evidence section.

### H1: The simulator systematically underestimates latency for memory-bound steps

> Decode-heavy workloads should show consistent negative prediction bias. The magnitude of the bias should be proportional to the fraction of step time spent memory-bound. Correcting for achievable (not theoretical) memory bandwidth should reduce this bias.

**How to test**: Run decode-only workloads at multiple batch sizes. Compare predicted vs. measured TPOT. The error should be directionally consistent (always underestimating) and larger at small batch sizes where steps are more memory-bound.

**Accept criterion**: Decode-only MAPE improves by ≥5pp after correction.

<details>
<summary><b>Mechanism and evidence</b></summary>

The model assumes GPUs sustain their datasheet peak HBM bandwidth. In practice, DRAM refresh cycles, bank conflicts, ECC overhead, and memory controller scheduling reduce sustained bandwidth to ~80% of theoretical peak. This is a well-known hardware property — NVIDIA STREAM benchmarks on H100 show 2650-2750 GB/s sustained against 3350 GB/s peak.

- BLIS uses raw peak: `3.35e12` bytes/s — [`sim/roofline_step_v2.go:132`](sim/roofline_step_v2.go#L132)
- InferSim applies 0.80 efficiency on every GPU it supports: H20 ([line 23](https://github.com/alibaba/InferSim/blob/main/hardware/gpu.py#L23)), H800 ([line 36](https://github.com/alibaba/InferSim/blob/main/hardware/gpu.py#L36)), H200 ([line 49](https://github.com/alibaba/InferSim/blob/main/hardware/gpu.py#L49)), GB200 ([line 59](https://github.com/alibaba/InferSim/blob/main/hardware/gpu.py#L59))
- Impact: 22% underestimate on memory-bound steps (`3350/2744 = 1.221`)

</details>

### H2: The simulator underestimates step time by a fixed additive amount independent of compute workload

> Across all workloads, there should be a constant positive residual (measured - predicted) per step. This residual should be roughly the same magnitude regardless of model size, batch size, or prefill/decode mix. It represents time spent outside GPU compute — scheduling, tensor preparation, output processing.

**How to test**: Across multiple workloads, compute residual = measured_step_time - predicted_step_time. The residual should cluster around a fixed value (not scale with FLOPs or memory). Regress residual against batch composition — the intercept should be large (ms-scale) and the slope near zero.

**Accept criterion**: TPOT MAPE improves by ≥5pp after adding per-step overhead.

<details>
<summary><b>Mechanism and evidence</b></summary>

The model captures GPU kernel execution time but misses the CPU-side overhead of the vLLM scheduler loop: Python-level batch formation, block table allocation, CUDA graph selection, input tensor preparation, output detokenization, and scheduler state updates. These are Python operations that take 3-30ms per step — 100-600× larger than the 50μs kernel launch latency BLIS currently models.

- BLIS overhead: `TOverheadMicros: 50.0` — [`hardware_config.json:6`](hardware_config.json#L6)
- InferSim decode overhead: `tpot += 5` (5ms) — [`models/model.py:229`](https://github.com/alibaba/InferSim/blob/main/models/model.py#L229)
- InferSim prefill overhead: `ttft += 30` (30ms) — [`models/model.py:177`](https://github.com/alibaba/InferSim/blob/main/models/model.py#L177)

</details>

### H3: Prediction accuracy degrades when the simulator queries the MFU database with GEMM shapes that differ from those actually executed by the GPU

> For models with grouped-query attention (GQA), the simulator should show larger GEMM-related prediction error than for models with standard multi-head attention — because GQA creates the largest shape mismatch between split (Q/K/V separate) and fused (QKV combined) projections. Aligning lookup shapes with execution shapes should reduce this error.

**How to test**: Compare error on a GQA model (e.g., Llama-3.1-8B, where K/V have 8 heads vs. Q's 32) against a non-GQA model. The GQA model should show larger GEMM-component error. After fusing the QKV lookup, the gap should close.

**Accept criterion**: GEMM-dominated workload MAPE improves by ≥2pp.

<details>
<summary><b>Mechanism and evidence</b></summary>

vLLM fuses Q, K, V projections into a single `qkv_proj` GEMM with output dimension `(num_heads + 2 * num_kv_heads) * head_dim`. The MFU database was benchmarked with these fused shapes. BLIS queries the database with three separate shapes (Q, K, V individually), which don't match any benchmarked entry and which have different MFU characteristics — smaller GEMMs achieve lower MFU due to insufficient SM parallelism.

- BLIS: 4 separate lookups per layer (Q, K, V, O) — [`sim/roofline_step_v2.go:54-68`](sim/roofline_step_v2.go#L54)
- InferSim: 2 lookups per layer (fused QKV, O) — [`flops/flops.py:9-20`](https://github.com/alibaba/InferSim/blob/main/flops/flops.py#L9)
- For Llama-3.1-8B: fused shape `(bs, 4096, 6144)` vs. split K/V shapes `(bs, 4096, 1024)`

</details>

### H4: The simulator underestimates latency for workloads where different components within a layer are bottlenecked by different resources

> For decode steps with large KV caches (where attention is memory-bound) but small batch sizes (where GEMMs are also memory-bound but with different operational intensity), the aggregate roofline should underpredict more than it does for uniformly compute-bound or uniformly memory-bound steps. The error should increase as the operational intensities of different components diverge.

**How to test**: Vary KV cache length while holding batch size fixed. At the crossover point where some components switch from compute-bound to memory-bound, prediction error should spike under the aggregate model but remain stable under per-component modeling.

**Accept criterion**: Per-step prediction variance decreases; no systematic error spike at bottleneck transitions.

<details>
<summary><b>Mechanism and evidence</b></summary>

The roofline model `T = max(W/P, Q/B)` assumes a single operational intensity. BLIS applies this globally — total FLOPs and total bytes across all components, then one `max`. InferSim applies it per-component within each layer (attention compute vs. KV load), then sums across layers. The aggregate approach averages away per-component bottleneck transitions.

- InferSim: per-component roofline [`layers/attn.py:25-38`](https://github.com/alibaba/InferSim/blob/main/layers/attn.py#L25), per-layer sum [`models/model.py:175`](https://github.com/alibaba/InferSim/blob/main/models/model.py#L175)
- BLIS: aggregate `math.Max` at [`sim/roofline_step_v2.go:275-276, 299`](sim/roofline_step_v2.go#L275)
- Theory: Williams et al., "Roofline: An Insightful Visual Performance Model", CACM 2009

</details>

### H5: Mixed-batch prediction error correlates with the prefill/decode token ratio

> For workloads with mixed prefill+decode steps, prediction error should vary systematically with the ratio of prefill to decode tokens in each step. Steps that are mostly prefill or mostly decode should be more accurate; steps near 50/50 should be least accurate. This is because a weighted-average blend is most wrong when both components contribute equally.

**How to test**: Bucket mixed-batch steps by prefill/decode ratio. Plot MAPE per bucket. The error should correlate with ratio distance from the extremes. After replacing the weighted average with an additive model, the correlation should disappear.

**Accept criterion**: Mixed-batch MAPE improves by ≥3pp; error no longer correlates with prefill/decode ratio.

<details>
<summary><b>Mechanism and evidence</b></summary>

In vLLM with chunked prefill, a mixed step executes as: (1) one fused GEMM over the concatenated prefill+decode batch for all linear projections, (2) separate FlashAttention kernels for prefill and decode tokens. The correct time model is additive: `GEMM(totalBatch) + PrefillAttn + DecodeAttn`. BLIS instead uses a weighted average of independent prefill and decode predictions with uncited magic constants.

- BLIS: `0.75*prefillTime + 0.25*decodeTime` (prefill-dominated), `0.35*prefillTime + 0.65*decodeTime` (decode-dominated) — [`sim/roofline_step_v2.go:288-291`](sim/roofline_step_v2.go#L288)
- These weights have no citation in InferSim, published literature, or vLLM documentation
- InferSim does not model mixed batches — this is a novel correction

</details>

### H6: Per-request prediction variance has artificial discontinuities at MFU database grid boundaries

> When batch size or sequence length crosses an MFU grid boundary, the simulator's predicted latency should jump discontinuously. This produces excess per-request prediction variance that is an artifact of the lookup method, not of real hardware behavior. Smoothing the MFU surface should reduce this variance.

**How to test**: Sweep batch size by 1 across a grid boundary. Plot predicted latency vs. batch size. Nearest-neighbor should produce a step function; interpolation should produce a smooth curve. Measure per-request prediction variance before/after.

**Accept criterion**: Per-request prediction variance decreases at grid boundaries.

<details>
<summary><b>Mechanism and evidence</b></summary>

Both InferSim and BLIS use nearest-neighbor MFU lookup without interpolation. GPU kernel performance varies smoothly with batch size and sequence length — adjacent configurations produce similar MFU. The discrete lookup creates artificial jumps.

- InferSim: independent 1D floor scans — [`mfu/mfu.py:30-52`](https://github.com/alibaba/InferSim/blob/main/mfu/mfu.py#L30)
- BLIS: 2D Euclidean distance with floor preference — `sim/mfu_database.go`
- Bilinear interpolation is a standard numerical technique that better captures the physically smooth MFU surface

</details>

---

## 2. The Tiered Accuracy Model (TAM)

TAM organizes corrections by calibration cost. Each tier is independently testable; improvements are cumulative.

### Tier 0: Zero-Config (No Calibration Required)

Corrections derived entirely from hardware physics, execution semantics, or standard numerical techniques. Zero fitted parameters.

| Hypothesis | What the simulator gets wrong | Accept criterion |
|------------|------------------------------|------------------|
| H1 | Underestimates memory-bound step latency | Decode-only MAPE ≥5pp better |
| H2 | Missing per-step scheduling overhead | TPOT MAPE ≥5pp better |
| H3 | MFU lookups use wrong GEMM shapes | GEMM-workload MAPE ≥2pp better |
| H4 | Aggregate roofline masks bottleneck transitions | Per-step variance decreases |
| H5 | Weighted-average mixed-batch model | Mixed MAPE ≥3pp better; no ratio correlation |
| H6 | MFU grid-boundary discontinuities | Per-request variance decreases at boundaries |

**Overfitting risk**: None for H1, H3, H4, H5 (zero fitted parameters). H2 uses a sweep but selects against held-out data. H6 is a standard numerical technique.

### Tier 1: One-Trace Calibration

> After Tier 0, the remaining prediction error should be dominated by version-specific vLLM overhead. One GuideLLM trace (≥100 requests) should be sufficient to extract this residual.

**Protocol**:
1. Run GuideLLM against real vLLM (any workload, ≥100 requests)
2. Run BLIS with Tier 0 corrections, overhead = floor
3. Match request pairs via calibration framework ([`sim/workload/calibrate.go`](sim/workload/calibrate.go))
4. Compute residual = measured − predicted per request
5. Fit: `residual_ms = a * numDecode + b * numPrefill + c * log2(1 + totalBatch) + d`
6. Cross-validate on 80/20 split

**Overfitting guard**: Validation MAPE > 2× training MAPE → reject, fall back to Tier 0.

**Staleness detection**: Residual MAPE > 30% on new trace → flag as stale.

### Tier 2: Hardware Characterization

> Tier 0 uses a single bandwidth efficiency factor for all memory access patterns. In reality, sequential reads (model weights) and scattered reads (PagedAttention KV cache) achieve different fractions of peak bandwidth. Per-pattern measurements should improve accuracy for KV-cache-heavy workloads.

Automated micro-benchmarks measuring bandwidth efficiency under three access patterns: sequential, scattered, and mixed. Replaces the uniform factor with measured per-pattern values.

---

## 3. Why This Works: Theoretical Grounding

### 3.1 Roofline Model Correctness

The roofline model `T = max(W/P, Q/B)` (Williams et al., CACM 2009) holds when the workload has a single operational intensity `W/Q`. H4 addresses the granularity at which this assumption holds — per-component within a layer, not aggregate across layers. H1 addresses the bandwidth parameter `B` — effective, not theoretical.

### 3.2 MFU as the Critical Accuracy Determinant

InferSim's core insight: *"The accuracy of the MFU directly determines the accuracy of the simulator results."* H3 ensures the MFU database is queried with the shapes that were actually benchmarked and executed. H6 smooths the lookup surface to reduce artificial noise.

### 3.3 Compute vs. Scheduling

A vLLM step consists of two sequential phases: GPU compute (GEMMs + attention) and CPU scheduling (Python-level batch formation, block allocation, detokenization). These are fundamentally different phenomena with different scaling. H2 addresses the missing scheduling component.

---

## 4. Addressing Reviewer Concerns

### 4.1 Accuracy

Each hypothesis has an explicit accept criterion in the table above. Corrections are applied incrementally with regression gates: if any metric worsens by >1pp, the correction is reverted. Cumulative expected improvement: 15-25pp at Tier 0.

### 4.2 Generalization Across Workloads

Tier 0 corrections are workload-agnostic: H1 addresses hardware physics, H3 addresses execution semantics, H4 addresses modeling theory, H5 addresses framework behavior, H2 addresses framework overhead, H6 addresses numerical technique. No correction is fitted to a specific workload. Cross-validation on held-out distributions (short vs. long context) required before acceptance.

### 4.3 Generalization Across vLLM Configurations

Tier 0 is vLLM-agnostic. Tier 1 adapts via one trace. Different vLLM configs produce different overhead residuals, captured by Tier 1 fitting.

### 4.4 Generalization Across LLMs

All corrections derive from model architecture parameters and hardware physics. Cross-validation on a different architecture family (e.g., Qwen2.5-7B) required before acceptance.

### 4.5 Across Hardware Types

The bandwidth efficiency phenomenon (H1) is universal for HBM — InferSim applies it across four GPU generations ([`hardware/gpu.py`](https://github.com/alibaba/InferSim/blob/main/hardware/gpu.py)). Tier 2 provides measurement for unlisted GPUs.

### 4.6 Ease of Use

| Tier | Setup | Accuracy Target | Recalibration |
|------|-------|----------------|---------------|
| 0 | None | < 20% E2E MAPE | Never |
| 1 | One GuideLLM trace | < 15% E2E MAPE | When vLLM version changes |
| 2 | Automated micro-benchmarks | < 12% E2E MAPE | When GPU hardware changes |

### 4.7 vLLM Version Resilience

- **Tier 0**: Immune — hardware/model properties.
- **Tier 1**: Version-tagged coefficients with staleness detection.
- **Tier 2**: GPU-intrinsic, unaffected by vLLM.

### 4.8 Overheads

Runtime: Zero or negligible (~50ns per interpolated MFU lookup). All corrections are O(1) modifications.

### 4.9 Reproducibility

Tier 0: Fully deterministic. Tier 1: Deterministic given same trace + seed (INV-6). All tiers: BLIS hypothesis workflow with `run.sh`, `analyze.py`, `FINDINGS.md`, 3-5 seeds.

---

## 5. Limitations and Open Questions

1. **MFU database coverage**: Accuracy is bounded by MFU data quality. Missing entries near target shapes degrade both nearest-neighbor and interpolation.

2. **InferSim as reference**: TAM aligns with InferSim's choices. InferSim itself has 4-15% error — we can approach but likely not exceed it without novel contributions.

3. **Mixed-batch validation**: H5 is a novel contribution (InferSim doesn't model mixed batches). The behavioral claim derives from vLLM's execution model but requires empirical validation.

4. **Overhead functional form**: H2 assumes scheduling overhead is roughly fixed per step. In reality it may scale weakly with batch size. Tier 1's fit captures some scaling; the flat assumption must be verified experimentally.

5. **Non-NVIDIA GPUs**: The bandwidth efficiency constant is validated only for NVIDIA HBM GPUs in InferSim. Tier 2 addresses other hardware through direct measurement.

---

## References

| Reference | Citation |
|-----------|----------|
| InferSim | [github.com/alibaba/InferSim](https://github.com/alibaba/InferSim) — Alimama AI Infra Team, Alibaba |
| InferSim BW factor | [`hardware/gpu.py:23,36,49,59`](https://github.com/alibaba/InferSim/blob/main/hardware/gpu.py#L36) |
| InferSim overheads | [`models/model.py:177,229`](https://github.com/alibaba/InferSim/blob/main/models/model.py#L177) |
| InferSim attention | [`layers/attn.py:25-38,86-94`](https://github.com/alibaba/InferSim/blob/main/layers/attn.py#L25) |
| InferSim FLOPs | [`flops/flops.py:9-20`](https://github.com/alibaba/InferSim/blob/main/flops/flops.py#L9) |
| InferSim MFU lookup | [`mfu/mfu.py:30-52,76-84`](https://github.com/alibaba/InferSim/blob/main/mfu/mfu.py#L30) |
| InferSim per-layer sum | [`models/model.py:175`](https://github.com/alibaba/InferSim/blob/main/models/model.py#L175) |
| InferSim causal mask | [`layers/attn.py:86`](https://github.com/alibaba/InferSim/blob/main/layers/attn.py#L86) — `/1.8` factor |
| BLIS v2 step time | [`sim/roofline_step_v2.go`](sim/roofline_step_v2.go) |
| BLIS v2 peak BW | [`sim/roofline_step_v2.go:132`](sim/roofline_step_v2.go#L132) |
| BLIS v2 Q/K/V/O GEMMs | [`sim/roofline_step_v2.go:54-68`](sim/roofline_step_v2.go#L54) |
| BLIS v2 mixed weights | [`sim/roofline_step_v2.go:288-291`](sim/roofline_step_v2.go#L288) |
| BLIS hardware config | [`hardware_config.json:6,18-19`](hardware_config.json#L6) |
| BLIS calibration | [`sim/workload/calibrate.go`](sim/workload/calibrate.go) |
| Roofline model | Williams et al., "Roofline: An Insightful Visual Performance Model", CACM 2009 |
| FlashAttention | Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention", NeurIPS 2022 |
| PagedAttention | Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention", SOSP 2023 |
| vLLM | Kwon et al., [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
