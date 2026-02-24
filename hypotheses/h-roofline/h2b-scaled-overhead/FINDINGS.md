# H2b: Model-Scaled Scheduling Overhead — Per-Layer/TP Formula

**Status:** Partial confirmation
**Resolution:** Partial confirmation with criterion 2 failure — the scaling direction is correct and dramatically outperforms fixed overhead, but the 100μs/layer base constant overshoots for 7B at TP=1
**Family:** Structural model (does model-scaled scheduling overhead explain the residual after H1?)
**VV&UQ:** Validation (comparing simulator predictions against real-world measurements)
**Tier:** 0 (zero-config correction)
**Type:** Deterministic (same config = identical output per INV-6)
**Date:** 2026-02-24
**Rounds:** 1

## Hypothesis

> "Per-step scheduling overhead scales with `num_hidden_layers / tp` — because the dominant CPU-side scheduler work per step is block-table management and per-layer tensor dispatch, both O(num_layers). A model-scaled overhead should produce uniform accuracy improvement across model sizes, unlike the fixed overhead (H2) which overshoots small models and undershoots large ones."

**Diagnostic:** If this fails while H2 also failed, it indicates that the residual error after H1 is not explained by scheduling overhead at all — pointing to compute-side corrections (H3 GEMM shapes, H4 per-component roofline).

## Experiment Design

**Classification:** Deterministic validation against ground truth

**ED compliance:**
- **ED-1 (Single dimension):** Only the overhead formula varies. H1 BW correction held constant (0.82) across all three configs.
- **ED-2 (Vanish rate):** For 7B at TP=4, the scaled overhead is only 800μs — the effect is minimal. For 70B at TP=4, it's 2000μs. The effect scales proportionally with model complexity.
- **ED-3 (Preconditions):** Script validates `model_configs/`, `bench_data/`, `eval/ground_truth/` exist. Model config.json parsed for `num_hidden_layers`.
- **ED-4 (Seed independence):** Deterministic, seed=42.
- **ED-5 (Reproducibility):** Self-contained `run.sh`.
- **ED-6 (Config diff):** Reference: `hypotheses/h-roofline/h2-scheduling-overhead/run.sh` — baseline matches H2's baseline. Fixed config matches H2's treatment. Scaled config differs: per-experiment `TOverheadMicros` computed as `100 × num_layers / tp`.

**Three configurations compared:**

| Config | TOverheadMicros | PrefillOverheadMicros | MixedPrefillOverheadMicros | BwEfficiencyFactor |
|--------|----------------|-----------------------|---------------------------|-------------------|
| Baseline (H1-only) | 0 | 0 | 0 | 0.82 |
| Fixed (H2) | 5000 | 30000 | 15000 | 0.82 |
| Scaled (H2b) | `100 × layers / tp` | `500 × layers / tp` | `250 × layers / tp` | 0.82 |

**Per-experiment computed overheads:**

| Model | Layers | TP | Decode | Prefill | Mixed |
|-------|--------|---|--------|---------|-------|
| Llama2-7B | 32 | 1 | 3200μs | 16000μs | 8000μs |
| Llama2-7B | 32 | 2 | 1600μs | 8000μs | 4000μs |
| Llama2-7B | 32 | 4 | 800μs | 4000μs | 2000μs |
| CodeLlama-34B | 48 | 2 | 2400μs | 12000μs | 6000μs |
| Llama2-70B | 80 | 4 | 2000μs | 10000μs | 5000μs |
| Qwen3-14B | 40 | 1 | 4000μs | 20000μs | 10000μs |
| Qwen3-14B | 40 | 2 | 2000μs | 10000μs | 5000μs |
| Qwen2.5-7B | 28 | 1 | 2800μs | 14000μs | 7000μs |

**Accept criteria:**
1. TPOT MAPE improves by >=3pp vs H1-only baseline across all 13 experiments
2. No single experiment worsens by >5pp vs H1-only
3. Scaled overhead produces lower aggregate TPOT MAPE than fixed-5ms overhead (H2)

## Results

### Per-Experiment TPOT MAPE (3-way comparison)

| Experiment | Type | BL TPOT | FX TPOT | SC TPOT | BL→SC | FX→SC |
|---|---|---|---|---|---|---|
| jan30-llama2-7b-tp1-chatsweep | chatsweep | 22.4% | 62.8% | 30.5% | **-8.2pp** | +32.3pp |
| jan30-llama2-7b-tp1-codesweep | codesweep | 32.1% | 49.3% | 15.2% | +16.9pp | +34.1pp |
| jan30-llama2-7b-tp2-chatsweep | chatsweep | 44.6% | 78.1% | 7.3% | +37.3pp | +70.8pp |
| jan30-llama2-7b-tp2-codesweep | codesweep | 51.5% | 71.2% | 18.7% | +32.8pp | +52.6pp |
| jan30-llama2-7b-tp4-chatsweep | chatsweep | 62.5% | 109.5% | 37.4% | +25.1pp | +72.1pp |
| jan30-llama2-7b-tp4-codesweep | codesweep | 68.5% | 118.5% | 47.7% | +20.7pp | +70.8pp |
| 20260210-codellama-34b-tp2-chatsweep | chatsweep | 13.8% | 22.1% | 3.3% | +10.5pp | +18.7pp |
| 20260210-codellama-34b-tp2-codesweep | codesweep | 26.1% | 6.1% | 11.0% | +15.1pp | -4.9pp |
| 20260210-llama2-70b-tp4-chatsweep | chatsweep | 20.4% | 11.7% | 7.6% | +12.8pp | +4.1pp |
| 20260210-llama2-70b-tp4-codesweep | codesweep | 32.4% | 3.9% | 21.2% | +11.1pp | -17.3pp |
| 20260210-qwen3-14b-tp1-codesweep | codesweep | 26.8% | 13.9% | 5.0% | +21.8pp | +8.9pp |
| 20260210-qwen3-14b-tp2-chatsweep | chatsweep | 36.6% | 29.8% | 10.3% | +26.4pp | +19.6pp |
| dec17-tp1-qwen7-summarization | summarization | 31.3% | 42.0% | 9.1% | +22.3pp | +32.9pp |

### Aggregate by Model Family

| Model Family | N | BL TPOT | FX TPOT | SC TPOT | BL→SC | FX→SC |
|---|---|---|---|---|---|---|
| Llama2-7B | 6 | 46.9% | 81.6% | 26.1% | +20.8pp | +55.4pp |
| Qwen2.5-7B | 1 | 31.3% | 42.0% | 9.1% | +22.3pp | +32.9pp |
| Qwen3-14B | 2 | 31.7% | 21.9% | 7.6% | +24.1pp | +14.2pp |
| CodeLlama-34B | 2 | 20.0% | 14.1% | 7.2% | +12.8pp | +6.9pp |
| Llama2-70B | 2 | 26.4% | 7.8% | 14.4% | +12.0pp | -6.6pp |

### Overall Aggregates

| Metric | Baseline (H1) | Fixed (H2) | Scaled (H2b) | BL→SC | FX→SC |
|---|---|---|---|---|---|
| **TPOT MAPE** | **36.1%** | **47.6%** | **17.3%** | **+18.8pp** | **+30.4pp** |
| **E2E MAPE** | **28.8%** | **70.9%** | **18.6%** | **+10.2pp** | **+52.3pp** |

### Signed Error (Bias Direction)

| Experiment | BL TPOT | FX TPOT | SC TPOT | What happened |
|---|---|---|---|---|
| Llama2-7b TP=1 chat | -22.4% | +62.8% | +30.5% | Scaled still overshoots: 3.2ms on ~5ms step |
| Llama2-7b TP=1 code | -32.1% | +49.3% | +15.2% | Scaled overpredicts mildly |
| Llama2-7b TP=2 | -44% to -52% | +71% to +78% | -7% to -19% | Near-correct — slight underprediction |
| Llama2-7b TP=4 | -63% to -69% | +110% to +119% | -37% to -48% | Better but still underpredicts (0.8ms on ~1ms step) |
| CodeLlama-34b TP=2 | -14% to -26% | +6% to +22% | +3% to -11% | Near-correct |
| Llama2-70b TP=4 | -20% to -32% | -4% to +12% | -8% to -21% | Slight underprediction |
| Qwen3-14b | -27% to -37% | +14% to +30% | +5% to -10% | Near-correct |
| Qwen2.5-7b TP=1 | -31.3% | +42.0% | +9.1% | Near-correct |

### Accept Criteria Evaluation

1. **TPOT MAPE improvement >= 3pp vs baseline: PASS (+18.8pp)**
2. **No experiment worsens by >5pp: FAIL** — jan30-llama2-7b-tp1-chatsweep worsened by 8.2pp (22.4% → 30.5%)
3. **Scaled TPOT MAPE < Fixed TPOT MAPE: PASS** — 17.3% vs 47.6% (+30.4pp better)

## Root Cause Analysis

### Why model-scaled overhead works dramatically better than fixed

The core issue identified in H2 was that InferSim's fixed 5ms overhead catastrophically overshoots for small/fast models (7B at TP=4: 5ms added to ~1ms step = 5× overshoot) while being approximately correct for large models. The model-scaled formula `100 × layers / tp` naturally produces:
- 0.8ms for 7B TP=4 (vs 5ms fixed = 6× reduction)
- 3.2ms for 7B TP=1 (vs 5ms fixed = 1.6× reduction)
- 2.0ms for 70B TP=4 (vs 5ms fixed = 2.5× reduction)
- 2.4ms for 34B TP=2 (vs 5ms fixed = 2.1× reduction)

This range (0.8ms – 4.0ms) is much closer to the observed residual scale (~2-3ms) than the flat 5ms.

### Why criterion 2 fails

The one failing experiment is 7B at TP=1 chatsweep, where the scaled overhead (3.2ms) is too high relative to the actual decode step time (~5ms). The baseline (no overhead) already underpredicts by 22.4%, but adding 3.2ms flips the bias to overprediction (+30.5%). The 100μs/layer constant is analytically estimated — the true per-layer scheduler cost at TP=1 may be closer to 50-70μs/layer.

Importantly, this is a **single experiment failure** in the specific regime where:
- The model is small (7B, 32 layers)
- TP is low (1 — no parallelism reduction)
- The workload is decode-heavy (chatsweep, output=215 tokens)
- So the overhead accumulates across many decode steps

For the same model at TP=2, the formula works well (7.3% TPOT MAPE, improved from 44.6%). The issue is specific to the combination of small model + low TP + decode-heavy workload.

### Why 7B TP=4 still has high error despite improvement

The 7B TP=4 experiments still show 37-48% TPOT MAPE even with scaled overhead. The signed errors show these are **still underpredicting** (-37% to -48%). The scaled overhead of 0.8ms is small compared to the baseline error. This regime is where the step time is ~1ms and the simulator has fundamental modeling gaps beyond scheduling overhead — likely H3 (GEMM shape mismatch) or H4 (per-component roofline). The overhead correction helps (reduced from 63-69% to 37-48%) but is insufficient alone.

### Why Llama2-70B shows less improvement than expected

The 70B experiments at TP=4 went from 20.4%/32.4% baseline to 7.6%/21.2% scaled. The 2ms overhead brings chatsweep close to correct (7.6% MAPE), but the codesweep experiment (21.2%) still has significant error. For codesweep (prefill-heavy, output=28 tokens), the prefill overhead matters more than decode overhead, and the 10ms prefill overhead may be too large. The fixed overhead actually performed better for 70B codesweep (3.9% vs 21.2%), suggesting 70B's residual is closer to 5ms decode + lower prefill overhead.

### Code path

The overhead is applied at `sim/roofline_step_v2.go:317-337`. `TOverheadMicros` is applied per step, then scaled logarithmically by batch size (line 322-326). At synchronous rate (batch=1), the scaling factor is 1.0 (no amplification), so the raw value is added directly to each step.

## Devil's Advocate

**Arguing this should be "Confirmed":**

Criterion 2 (no experiment >5pp worse) is conservative — it was designed to catch the catastrophic overshooting seen in H2 (where 8 of 13 experiments worsened, some by 40-50pp). Here, only 1 of 13 worsens, and by only 8.2pp. The aggregate improvement is enormous: 18.8pp TPOT, 10.2pp E2E. If we relaxed criterion 2 to "no experiment worsens by >10pp" or used median instead of worst-case, this passes on all criteria. The 100μs/layer constant could be tuned down to 70-80μs to avoid the 7B-TP1 overshoot, but that would be fitting to ground truth — which violates the Tier 0 constraint.

**Arguing this should be "Refuted":**

The formula produces the wrong sign for 7B TP=1 (overpredicts instead of underpredicts). Any overhead formula that can flip the bias direction for a common configuration (7B at TP=1 is realistic for inference) is not reliably zero-config. Additionally, the per-layer scaling assumption is questionable — block table management may not truly scale linearly with layers. The true functional form might be `base + α * sqrt(num_layers)` or `base + α * log(num_layers)`.

## Findings Classification

| Finding | Type | Action |
|---|---|---|
| Model-scaled overhead (100μs/layer/tp) reduces TPOT MAPE from 36.1% to 17.3% | Confirmation | Apply as Tier 0 correction with the computed per-experiment overheads |
| Formula dramatically outperforms fixed InferSim overhead (+30.4pp better) | Confirmation | Model-scaled overhead replaces fixed overhead as the recommended approach |
| 7B TP=1 chatsweep worsens by 8.2pp (overprediction) | Partial refutation | The 100μs/layer constant may be too high; a sweep in 10μs increments (50-150μs range) or a secondary scaling by TP could improve this regime |
| 7B TP=4 still has 37-48% MAPE despite improvement | Open question | Remaining error in this regime likely requires H3 (GEMM shapes) or H4 (per-component roofline) |
| 70B codesweep benefits less from scaled than from fixed overhead | Open question | Prefill overhead scaling may need different base constant than decode |

## Standards Audit

- [x] Any violations of existing rules? None found.
- [x] Any new rules needed? None.
- [x] Any new invariants needed? None.
- [x] Any existing rules/invariants confirmed? INV-6 (determinism): re-running produces identical output.

## Scope and Limitations

**Operating point tested:**
- GPU: H100 only
- Models: 5 dense model families (7B, 14B, 28B, 34B, 70B), no MoE
- TP: 1, 2, 4
- Rate: Synchronous
- Base constants: 100μs/layer decode, 500μs/layer prefill (single analytical estimate — no sweep)

**What was NOT tested:**
- A sweep of `base_per_layer_us` (50-150μs in 10μs steps) to find optimal value
- Non-linear scaling formulas (sqrt, log)
- Separate scaling for prefill vs decode base constants
- Interaction with H3 (GEMM shapes) or H4 (per-component roofline) — cumulative effects
- Non-synchronous rates (queuing effects amplify overhead per step)

**Generalizability:**
- The formula generalizes across 5 model families and 3 TP values with 12/13 experiments improving
- The one failure (7B TP=1 chatsweep) is in a regime where the overhead is a large fraction of step time
- The formula should generalize to other dense models with known `num_hidden_layers`

**Uncertainty quantification:**
- Not applicable — single deterministic run per configuration, no sweep or cross-validation

## Evidence Quality

| Metric | Value | Confidence |
|---|---|---|
| TPOT MAPE change (overall) | +18.8pp (36.1% → 17.3%) | High — clear improvement across 12 of 13 experiments |
| E2E MAPE change (overall) | +10.2pp (28.8% → 18.6%) | High — improvement across 11 of 13 experiments |
| Scaled vs Fixed comparison | +30.4pp TPOT, +52.3pp E2E | High — scaled dramatically outperforms fixed |
| Criterion 2 failure | 1 of 13 experiments worsens by 8.2pp | Medium — marginal failure, specific to 7B TP=1 decode-heavy |
| Best model family (Qwen3-14B) | 31.7% → 7.6% TPOT MAPE | High |
| Worst model family (7B TP=4) | 65.5% → 42.6% TPOT MAPE | Medium — improved but still high due to non-overhead errors |

## Implications for Users

1. **Model-scaled overhead is the recommended Tier 0 approach.** Use `TOverheadMicros = 100 × num_hidden_layers / tp` and `PrefillOverheadMicros = 500 × num_hidden_layers / tp` in `hardware_config.json`. This requires per-model hardware configs (not a single shared config).

2. **Overall accuracy after H1+H2b:** TPOT MAPE drops from the raw baseline of ~47.7% to **17.3%** — within striking distance of the Tier 0 target (<20%).

3. **For 7B models at TP=1**, the overhead may overshoot. Users can reduce the base constant to 70μs/layer or skip the overhead for this specific regime.

4. **This correction stacks with H1 (BW efficiency).** The cumulative effect is: raw → H1 → H1+H2b = 47.7% → 36.1% → 17.3% TPOT MAPE.

5. **Do NOT use InferSim's fixed 5ms/30ms values.** The model-scaled approach is 30pp better on TPOT MAPE.

## Reproducing

```bash
cd hypotheses/h-roofline/h2b-scaled-overhead && ./run.sh
```

Requires: `model_configs/`, `bench_data/`, `eval/ground_truth/` directories populated.
