# Idea 3: Hierarchical Two-Stage Model — FINDINGS SUMMARY

## Status: REFUTED (Catastrophic Failure)

## Approach

Two-stage decomposition:
1. **Stage 1 (Delta Model)**: Train a single Ridge regression across ALL models on step-level residuals (step.duration_us - metadata_overhead), capturing batch-composition effects
2. **Stage 2 (Scale Model)**: Use metadata-derived overhead as the base, add Stage 1's learned delta

The goal was a single unified model that works across all 4 architectures without per-model calibration.

### Fatal Flaw

The delta Ridge model trained on residuals (`step_duration - metadata_overhead`) produced:
- **Intercept = -22,254** (huge negative — residuals are mostly negative because metadata overhead > step duration)
- **decode_coeff = 665.2** (enormous — compensating for negative intercept)
- **prefill_coeff = 32.3**

After combining with metadata overhead and clamping to physical bounds:
- beta0 clamped to minimum (1,000 μs) — metadata_overhead + (-22,254) is deeply negative
- beta2 = 665 — a single decode token adds 665μs, producing step times of 665 × 34 = 22,610μs for typical batches
- Result: Predictions 20-100× too high

## Results

### H1: Full Validation (All 10 Experiments)

| Experiment | E2E Error | GT (ms) | Pred (ms) | Status |
|------------|-----------|---------|-----------|--------|
| llama-2-7b-roleplay | 4,778% | 2,071 | 101,028 | FAIL |
| llama-2-70b-general | 7,186% | 5,321 | 387,667 | FAIL |
| llama-2-70b-hf-codegen | 3,769% | 4,605 | 178,185 | FAIL |
| llama-2-70b-roleplay | 2,116% | 4,562 | 101,073 | FAIL |
| mixtral-codegen | 3,711% | 4,675 | 178,173 | FAIL |
| mixtral-general | 7,594% | 5,039 | 387,655 | FAIL |
| mixtral-roleplay | 2,057% | 4,685 | 101,062 | FAIL |
| codellama-34b-general | 9,372% | 4,093 | 387,647 | FAIL |
| codellama-34b-codegen | 4,686% | 3,723 | 178,165 | FAIL |
| codellama-34b-roleplay | 2,654% | 3,670 | 101,054 | FAIL |
| **MEAN** | **4,792%** | | | **0/10 <10%** |

### H2: LOMO

| Held-Out Model | E2E Error (%) | Pass (<80%) |
|----------------|---------------|-------------|
| llama-2-7b | 8,941% | FAIL |
| codellama-34b | 2,968% | FAIL |
| llama-2-70b | 3,745% | FAIL |
| mixtral-8x7b | 5,778% | FAIL |
| **MEAN** | **5,358%** | **0/4 PASS** |

### H3: LOWO

Not meaningful — all results catastrophically wrong.

## Root Cause Analysis

The hierarchical two-stage approach fails because:

1. **Residual distribution is pathological**: `step_duration - metadata_overhead` is negative for 77.9% of steps (decode-only steps are 12-7000μs while metadata overhead is 9,741-18,921μs). Ridge fits a negative intercept to match the negative mean, producing unusable coefficients.

2. **BlackboxLatencyModel cannot represent the learned function**: The model needs `step_time = max(overhead, compute_delta)`, not `step_time = overhead + delta`. Adding a negative intercept to a positive overhead produces nonsense — the clamping to minimum bounds removes all useful signal.

3. **Cross-model pooling destroys scale**: The 3+ OOM variation across model step times (12μs to 250,000μs) means a single Ridge model cannot find coefficients that work for all scales. This was known from R1 (confirmed again).

## Lessons Learned

- **Do not train residual models when residuals are systematically negative** — the intercept will be large and negative, producing invalid coefficients
- **Cross-model step-level training confirmed failed for the 4th time** (R1, R2, R3, now R5). The only viable path is per-model coefficients or metadata-derived coefficients.
- **The BlackboxLatencyModel's additive form requires positive coefficients** — any approach producing negative intercepts is fundamentally incompatible
