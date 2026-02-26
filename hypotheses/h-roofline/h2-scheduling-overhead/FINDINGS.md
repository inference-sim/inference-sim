# H2: Scheduling Overhead Validation — Fixed Additive Per-Step Overhead

**Status:** Refuted
**Resolution:** Refuted — mechanism not plausible at InferSim scale
**Family:** Structural model (does a fixed scheduling overhead explain the residual after H1?)
**VV&UQ:** Validation (comparing simulator predictions against real-world measurements)
**Tier:** 0 (zero-config correction)
**Type:** Deterministic (same config = identical output per INV-6)
**Date:** 2026-02-24
**Rounds:** 1

## Hypothesis

> "The simulator underestimates step time by a fixed additive amount independent of compute workload — representing CPU-side vLLM scheduling overhead (~5ms decode, ~30ms prefill) that the roofline model does not capture."

**Diagnostic:** If MAPE worsens, it indicates InferSim's overhead values are calibrated to InferSim's own modeling gaps, not to the actual vLLM scheduling overhead as seen from BLIS's roofline v2 model.

## Experiment Design

**Classification:** Deterministic validation against ground truth

**ED compliance:**
- **ED-1 (Single dimension):** Only overhead parameters vary (`TOverheadMicros`, `PrefillOverheadMicros`, `MixedPrefillOverheadMicros`). H1 BW correction held constant at 0.82.
- **ED-2 (Vanish rate):** For large models (70B) where GPU compute time dominates, a 5ms overhead is a small relative addition. For small models at high TP (7B TP=4) where step time is ~1ms, the same 5ms is a 5x addition. The effect should be proportionally larger for fast models.
- **ED-3 (Preconditions):** Script validates `model_configs/`, `bench_data/`, `eval/ground_truth/` exist.
- **ED-4 (Seed independence):** Deterministic, seed=42.
- **ED-5 (Reproducibility):** Self-contained `run.sh`.
- **ED-6 (Config diff):** Reference: `hypotheses/h-roofline/h1-bw-efficiency/run.sh` — baseline config matches H1 treatment config.

**Configurations compared:**

- **Baseline:** H1 BW correction only
  - `{"H100": {"TFlopsPeak": 989.5, "BwPeakTBs": 3.35, "bwEfficiencyFactor": 0.82}}`

- **Treatment:** H1 + InferSim-scale overheads
  - `{"H100": {"TFlopsPeak": 989.5, "BwPeakTBs": 3.35, "bwEfficiencyFactor": 0.82, "TOverheadMicros": 5000, "prefillOverheadMicros": 30000, "mixedPrefillOverheadMicros": 15000}}`

**Overhead values from InferSim:**
- Decode: `tpot += 5` (5ms) — `models/model.py:229`
- Prefill: `ttft += 30` (30ms) — `models/model.py:177`
- Mixed: 15ms (interpolation, no InferSim reference)

**Accept criteria:**
1. TPOT MAPE improves by >=3pp across all 13 experiments
2. TTFT MAPE does not worsen by >1pp

## Results

### Per-Experiment MAPE Comparison

| Experiment | Type | BL TPOT | TR TPOT | Delta | BL E2E | TR E2E | Delta |
|---|---|---|---|---|---|---|---|
| jan30-llama2-7b-tp1-chatsweep | chatsweep | 22.4% | 62.8% | -40.5pp | 20.1% | 69.2% | -49.2pp |
| jan30-llama2-7b-tp1-codesweep | codesweep | 32.1% | 49.3% | -17.2pp | 17.7% | 93.6% | -75.9pp |
| jan30-llama2-7b-tp2-chatsweep | chatsweep | 44.6% | 78.1% | -33.4pp | 43.0% | 85.7% | -42.7pp |
| jan30-llama2-7b-tp2-codesweep | codesweep | 51.5% | 71.2% | -19.7pp | 41.2% | 127.2% | -86.1pp |
| jan30-llama2-7b-tp4-chatsweep | chatsweep | 62.5% | 109.5% | -47.0pp | 61.4% | 119.2% | -57.7pp |
| jan30-llama2-7b-tp4-codesweep | codesweep | 68.5% | 118.5% | -50.1pp | 61.8% | 191.7% | -130.0pp |
| 20260210-codellama-34b-tp2-chatsweep | chatsweep | 13.8% | 22.1% | -8.2pp | 11.3% | 26.6% | -15.3pp |
| 20260210-codellama-34b-tp2-codesweep | codesweep | 26.1% | 6.1% | **+20.0pp** | 10.7% | 34.0% | -23.3pp |
| 20260210-llama2-70b-tp4-chatsweep | chatsweep | 20.4% | 11.7% | **+8.7pp** | 18.0% | 15.8% | **+2.2pp** |
| 20260210-llama2-70b-tp4-codesweep | codesweep | 32.4% | 3.9% | **+28.5pp** | 18.2% | 21.5% | -3.3pp |
| 20260210-qwen3-14b-tp1-codesweep | codesweep | 26.8% | 13.9% | **+13.0pp** | 9.6% | 47.9% | -38.3pp |
| 20260210-qwen3-14b-tp2-chatsweep | chatsweep | 36.6% | 29.8% | **+6.8pp** | 34.8% | 35.1% | -0.3pp |
| dec17-tp1-qwen7-summarization | summarization | 31.3% | 42.0% | -10.6pp | 26.1% | 54.1% | -28.0pp |

### Aggregate by Workload Type

| Workload Type | N | BL TPOT | TR TPOT | Delta | BL E2E | TR E2E | Delta |
|---|---|---|---|---|---|---|---|
| chatsweep | 6 | 33.4% | 52.3% | -18.9pp | 31.4% | 58.6% | -27.2pp |
| codesweep | 6 | 39.6% | 43.8% | -4.3pp | 26.5% | 86.0% | -59.5pp |
| summarization | 1 | 31.3% | 42.0% | -10.6pp | 26.1% | 54.1% | -28.0pp |
| **OVERALL** | **13** | **36.1%** | **47.6%** | **-11.5pp** | **28.8%** | **70.9%** | **-42.1pp** |

### Bias Direction (Signed Error)

The baseline (H1 only) uniformly underpredicts (all negative). The treatment flips most experiments to overprediction:

| Model size | BL TPOT bias | TR TPOT bias | What happened |
|---|---|---|---|
| Llama2-7b TP=4 | -62% to -69% | +109% to +119% | Catastrophic overshoot: 5ms added to ~1ms step |
| Llama2-7b TP=1 | -22% to -32% | +49% to +63% | Overshoot: 5ms added to ~5ms step |
| CodeLlama-34b TP=2 | -14% to -26% | +6% to +22% | Near-correct for TPOT, still overshoots TTFT |
| Llama2-70b TP=4 | -20% to -32% | -4% to +12% | Best fit — 5ms overhead is right scale for ~13ms step |
| Qwen3-14b | -27% to -37% | +14% to +30% | Moderate overshoot |

### Accept Criteria Evaluation

1. **TPOT MAPE improvement >= 3pp: FAIL (-11.5pp, worsened)**
2. **TTFT MAPE does not worsen by >1pp: PASS (improved by 8.0pp)**

## Root Cause Analysis

### Why InferSim's overhead values fail for BLIS

InferSim's 5ms decode and 30ms prefill overheads are calibrated to InferSim's own latency model, not to the absolute vLLM scheduling overhead. InferSim's overhead terms absorb multiple modeling gaps simultaneously:

1. **MFU differences**: InferSim uses different MFU lookup tables and interpolation methods than BLIS's roofline v2. The overhead partially compensates for MFU estimation errors.
2. **Memory model differences**: InferSim's memory bandwidth calculation differs from BLIS's `calculateMemoryAccessBytes()` at `sim/roofline_step.go:70-149`. The overhead absorbs memory modeling discrepancies.
3. **Architecture-level differences**: InferSim applies overhead as a flat additive constant to the final TPOT/TTFT. BLIS applies it within the step time calculation at `sim/roofline_step_v2.go:321-337`, where it interacts with the batch-aware scaling formula.

The fundamental problem: **a fixed additive overhead is the wrong functional form for the residual error.** The H1 results show residuals of:
- ~2ms gap for 7B models at TP=1 (step time ~5ms, gap is ~40% of step)
- ~2ms gap for 34B models at TP=2 (step time ~12ms, gap is ~17% of step)
- ~3ms gap for 70B models at TP=4 (step time ~13ms, gap is ~24% of step)

The residual is not constant in absolute terms — it scales roughly with model size and inversely with TP. A constant 5ms overshoots for small/fast models and is approximately right for large models. This suggests the true overhead is either:
- Proportional to step time (multiplicative, not additive) — pointing to H3 (GEMM shape) or H4 (per-component roofline)
- A smaller constant (~1-2ms) plus a model-dependent component

### Experiments where overhead helped

The 5 experiments that improved (CodeLlama-34b codesweep, Llama2-70b both, Qwen3-14b both) are all larger models (>=14B params) where the baseline step time is 10-13ms. For these, the 5ms decode overhead represents a 38-50% addition, which partially compensates for the systematic underprediction. This is consistent with the overhead being approximately correct in scale for large models but catastrophically wrong for small ones.

### Why TTFT improved while TPOT worsened

TTFT improved (+8pp) because the baseline TTFT predictions were extremely low (1-17ms vs ground truth 11-179ms). The 30ms prefill overhead pushes predictions closer to ground truth for most experiments. However, E2E worsened (-42pp) because the decode overhead compounds across all output tokens: a 5ms overhead per decode step x 215 output tokens = 1075ms of excess latency per request for chatsweep, which vastly overshoots.

### Code path for overhead application

At `sim/roofline_step_v2.go:317-326`, the `TOverheadMicros` is applied once per step (not per token), then scaled logarithmically by batch size. In the synchronous ground truth benchmark (1 request at a time), the batch size is 1 for decode steps, so the scaling factor is 1.0 (no amplification). The raw 5000μs is added directly to each decode step's time.

For a chatsweep request with 215 output tokens, the simulator runs ~215 decode steps, each adding 5ms overhead = 1075ms total additional latency per request. Ground truth TPOT for Llama2-7b TP=1 is 6.2ms; the overhead alone (5ms) nearly equals the ground truth step time.

## Devil's Advocate

**Arguing this should be "Confirmed":**

The overhead did help for 5 of 13 experiments, and these are arguably the most production-relevant configurations (larger models, higher TP). The 7B-at-TP=4 configurations are toy setups rarely used in production. If we restrict the evaluation to models >= 14B, the overhead improves TPOT for 4 out of 7 experiments. The hypothesis that a fixed overhead exists is not wrong — the problem is the magnitude, not the mechanism. A sweep from 500μs to 3000μs might find a value that improves all experiments. The TAM doc itself recommended a sweep with train/holdout validation, which this experiment did not implement per the simplification decision.

## Findings Classification

| Finding | Type | Action |
|---|---|---|
| InferSim's 5ms/30ms overheads are too large for BLIS's roofline v2 | Refutation | Do not use InferSim's overhead values directly |
| The residual error after H1 is not a constant additive term — it scales with model size | Surprise | The error structure suggests multiplicative corrections (H3, H4) rather than additive overhead |
| 30ms prefill overhead improves TTFT predictions (+8pp) | Partial confirmation | Prefill overhead concept is valid but the magnitude needs calibration |
| Overhead helps for large models (>=14B) but catastrophically overshoots for small models | Open question | A model-size-dependent or sweep-based overhead calibration is needed (TAM Tier 1) |
| E2E MAPE worsened by 42pp due to overhead compounding across decode steps | Surprise | The per-step overhead accumulates linearly with output length, making E2E extremely sensitive to overhead magnitude |

## Standards Audit

- [x] Any violations of existing rules? None found.
- [x] Any new rules needed? None.
- [x] Any new invariants needed? None.
- [x] Any existing rules/invariants confirmed? INV-6 (determinism): re-running produces identical output.

## Scope and Limitations

**Operating point tested:**
- GPU: H100 only
- Models: 5 dense model families (7B, 14B, 34B, 70B), no MoE
- TP: 1, 2, 4
- Rate: Synchronous
- Overhead values: Only InferSim's fixed values (5ms/30ms/15ms) — no sweep performed

**What was NOT tested:**
- A sweep of overhead values (500μs to 8000μs in 1ms steps) with train/holdout split — this was the TAM-recommended approach but was simplified per design approval
- Model-size-dependent overhead (e.g., overhead proportional to num_layers)
- Separate decode-only vs prefill-only overhead calibration
- Interaction with H3 (GEMM shapes) or H4 (per-component roofline)

**Generalizability:**
- The refutation is specific to InferSim's exact overhead values (5ms/30ms). It does NOT refute the existence of scheduling overhead — only that InferSim's values are wrong for BLIS.
- A smaller overhead (1-2ms) might improve all experiments without overshooting.

**Uncertainty quantification:**
- Not applicable — single fixed-value test, no sweep or cross-validation.

## Evidence Quality

| Metric | Value | Confidence |
|---|---|---|
| TPOT MAPE change (overall) | -11.5pp (36.1% -> 47.6%, worsened) | High — clear degradation across 8 of 13 experiments |
| E2E MAPE change (overall) | -42.1pp (28.8% -> 70.9%, worsened) | High — catastrophic degradation |
| TTFT MAPE change (overall) | +8.0pp (improved) | Medium — prefill overhead direction is correct |
| Large-model TPOT improvement | 4 of 7 experiments >= 14B improved | Medium — suggests correct mechanism, wrong magnitude |

## Implications for Users

1. **Do NOT apply InferSim's 5ms/30ms overhead values to BLIS.** They are calibrated to InferSim's modeling gaps, not to actual vLLM scheduling overhead.

2. **The residual error after H1 is not fixed-additive.** It scales with model configuration. A constant overhead will always overshoot for small/fast models and undershoot for large/slow ones.

3. **If you need overhead correction**, use a much smaller value (~1-2ms) or defer to TAM Tier 1 (one-trace calibration) which can fit the overhead as a function of model characteristics.

4. **Prefill overhead is directionally correct.** TTFT predictions improve with a prefill overhead term. The magnitude should be calibrated per-model, not fixed at 30ms.

## Reproducing

```bash
cd hypotheses/h-roofline/h2-scheduling-overhead && ./run.sh
```

Requires: `model_configs/`, `bench_data/`, `eval/ground_truth/` directories populated.
