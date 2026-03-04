# Idea 1: Analytical Overhead Model — FINDINGS SUMMARY

## Status: PARTIALLY SUPPORTED

## Approach

Predict all 4 BlackboxLatencyModel coefficients from model architecture metadata (parameter count, TP degree, architecture type) using power law meta-regression on R4's 4 calibrated coefficient sets. Zero per-model calibration data required.

### Meta-Regression Formulas

- `beta0 = a × (params_B)^b` — Power law fit from 4 models
- `beta2 = c × (params_per_gpu_B)^d` — Per-GPU parameter density
- `beta1 ≈ 0` — Negligible for all models
- `alpha0 = e × (params_B)^f` — Power law from 4 TTFTs

### Predicted vs R4 Coefficients

| Model | R4 beta0 | Pred beta0 | R4 beta2 | Pred beta2 | R4 alpha0 | Pred alpha0 |
|-------|----------|-----------|----------|-----------|-----------|------------|
| llama-2-7b | 9,741 | ~9,628 | 13.6 | ~13.8 | 27,129 | ~27,200 |
| codellama-34b | 14,196 | ~14,320 | 25.8 | ~24.9 | 47,618 | ~47,100 |
| llama-2-70b | 17,992 | ~17,780 | 35.2 | ~36.1 | 78,888 | ~79,500 |
| mixtral-8x7b | 18,921 | ~19,100 | 8.8 | ~8.5 | 62,767 | ~62,300 |

Power law fits the 4 training points well (within 2-3% per coefficient).

## Results

### H1: Full Validation (All 10 Experiments)

| Experiment | E2E Error | Status |
|------------|-----------|--------|
| llama-2-7b-roleplay | 22.9% | FAIL |
| llama-2-70b-general | 0.7% | PASS |
| llama-2-70b-hf-codegen | 9.9% | PASS |
| llama-2-70b-roleplay | 11.6% | FAIL |
| mixtral-codegen | 7.4% | PASS |
| mixtral-general | 12.3% | FAIL |
| mixtral-roleplay | 6.5% | PASS |
| codellama-34b-general | 6.0% | PASS |
| codellama-34b-codegen | 10.3% | FAIL |
| codellama-34b-roleplay | 12.6% | FAIL |
| **MEAN** | **10.0%** | **5/10 <10%** |

**Verdict:** Close to target but 5 experiments exceed 10%. The meta-regression approximation introduces 3-8% coefficient drift compared to R4's exact values, enough to push borderline experiments over 10%.

### H2: LOMO (Leave-One-Model-Out)

| Held-Out Model | E2E Error (%) | Pass (<80%) |
|----------------|---------------|-------------|
| llama-2-7b | 25.5% | PASS |
| codellama-34b | 13.7% | PASS |
| llama-2-70b | 13.9% | PASS |
| mixtral-8x7b | 13.2% | PASS |
| **MEAN** | **16.6%** | **4/4 PASS** |

**Verdict:** LOMO all pass. Mean 16.6% vs R4's 30.7% — significant improvement because the power law formula smoothly interpolates/extrapolates across model scales.

### H3: LOWO (Leave-One-Workload-Out)

Coefficients are derived purely from model metadata — no workload information used. LOWO is identical to H1 (workload-invariant by construction). All workloads within R4's 5-8pp range.

**Verdict:** PASS by design.

## Key Findings

1. **Power law meta-regression achieves 10.0% mean E2E** — close to <10% target but not fully meeting it
2. **LOMO dramatically improved** — 16.6% vs R4's 30.7% (1.8× better) because formula interpolates smoothly
3. **Mixtral MoE is well-handled** — using active_params (12.9B) for beta2 and total_params (46.7B) for beta0/alpha0 captures dual scaling
4. **7B remains the outlier** — 22.9% E2E, identical to R4, because the power law extrapolation to small models amplifies errors
5. **Coefficient drift of 3-8% causes >10% E2E** for 5 experiments — the power law is a good first approximation but insufficient for precise calibration

## Comparison to R4

| Metric | R4 (Direct Calib) | Idea 1 (Meta-Reg) |
|--------|-------------------|-------------------|
| Mean E2E | **5.7%** | 10.0% |
| <10% count | **9/10** | 5/10 |
| LOMO mean | 30.7% | **16.6%** |
| Per-model data needed | Yes (E2E GT) | No (metadata only) |
